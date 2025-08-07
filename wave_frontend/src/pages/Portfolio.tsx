import React from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'

export const Portfolio: React.FC = () => {
  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Overview</CardTitle>
          <CardDescription>Your balances, positions, and performance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-12">
            <p className="text-gray-400 mb-2">Portfolio view coming soon</p>
            <p className="text-sm text-gray-500">Configure your exchange keys to view your portfolio</p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}